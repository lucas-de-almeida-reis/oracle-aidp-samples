--------------------------------------------------------------------------------------------
--  Preflight: can this user drive the Iceberg external table sync?
--
--  Run CONNECTED AS THE CANDIDATE `adw_user`, on ONE ADW, before putting it in adw_sync.yaml.
--  Every check maps to a statement the notebook's ADMINISTRATIVE connection actually issues.
--
--  Read-mostly and self-cleaning: it creates one throwaway user (ADW_SYNC_PREFLIGHT_TMP), a
--  throwaway table in your own schema, and an ACL entry on the bogus host `preflight.invalid`
--  - then removes all three. It never touches a real schema, the real Object Storage host, or
--  any existing registry.
--
--    sqlplus <user>/<pwd>@<tns>  @preflight_adw_user.sql
--
--  Every line prints PASS or FAIL. Any FAIL is a blocker: fix the grant, or use a user that
--  already carries it. FAILs also print the Oracle error, which names the missing privilege.
--------------------------------------------------------------------------------------------
SET SERVEROUTPUT ON SIZE UNLIMITED
SET FEEDBACK OFF
SET LINESIZE 200

DECLARE
   c_tmp_user  CONSTANT VARCHAR2(30) := 'ADW_SYNC_PREFLIGHT_TMP';
   c_tmp_tab   CONSTANT VARCHAR2(30) := 'ADW_SYNC_PREFLIGHT_REG';
   c_tmp_host  CONSTANT VARCHAR2(64) := 'preflight.invalid';
   n_fail      PLS_INTEGER := 0;

   PROCEDURE report(p_label VARCHAR2, p_ok BOOLEAN, p_err VARCHAR2 DEFAULT NULL) IS
   BEGIN
      IF p_ok THEN
         DBMS_OUTPUT.PUT_LINE(RPAD(p_label, 58, '.') || ' PASS');
      ELSE
         n_fail := n_fail + 1;
         DBMS_OUTPUT.PUT_LINE(RPAD(p_label, 58, '.') || ' FAIL');
         DBMS_OUTPUT.PUT_LINE('        ' || SUBSTR(p_err, 1, 160));
      END IF;
   END;

   -- Run one statement, report PASS/FAIL. p_ignore lets an "already exists" count as success.
   PROCEDURE try(p_label VARCHAR2, p_sql VARCHAR2, p_ignore PLS_INTEGER DEFAULT NULL) IS
   BEGIN
      EXECUTE IMMEDIATE p_sql;
      report(p_label, TRUE);
   EXCEPTION
      WHEN OTHERS THEN
         IF p_ignore IS NOT NULL AND SQLCODE = p_ignore THEN
            report(p_label, TRUE);
         ELSE
            report(p_label, FALSE, SQLERRM);
         END IF;
   END;

   PROCEDURE quietly(p_sql VARCHAR2) IS   -- best-effort cleanup, never reports
   BEGIN
      EXECUTE IMMEDIATE p_sql;
   EXCEPTION WHEN OTHERS THEN NULL;
   END;

BEGIN
   DBMS_OUTPUT.PUT_LINE('Connected as : ' || USER);
   DECLARE
      v_roles VARCHAR2(4000);
   BEGIN
      SELECT LISTAGG(role, ', ') WITHIN GROUP (ORDER BY role)
        INTO v_roles
        FROM session_roles;
      DBMS_OUTPUT.PUT_LINE('Roles        : ' || NVL(v_roles, '(none)'));
   EXCEPTION
      WHEN OTHERS THEN DBMS_OUTPUT.PUT_LINE('Roles        : (unreadable) ' || SQLERRM);
   END;
   DBMS_OUTPUT.PUT_LINE(RPAD('-', 66, '-'));

   ----------------------------------------------------------------------------------------
   -- 0. Session prep, exactly as the notebook's _prep() does it. Must come FIRST: the state
   --    cannot be changed once a transaction is open (ORA-12841), and without it the registry
   --    MERGE runs as parallel DML and the next statement hits ORA-12838.
   ----------------------------------------------------------------------------------------
   try('ALTER SESSION DISABLE PARALLEL DML', 'ALTER SESSION DISABLE PARALLEL DML');

   ----------------------------------------------------------------------------------------
   -- 1. User lifecycle: the job creates one ADW user per source schema and rotates its
   --    password on every run.
   ----------------------------------------------------------------------------------------
   quietly('DROP USER ' || c_tmp_user || ' CASCADE');
   try('CREATE USER',
       'CREATE USER ' || c_tmp_user || ' IDENTIFIED BY "Pf_' || DBMS_RANDOM.STRING('X', 12) || '1#"');
   try('ALTER USER ... IDENTIFIED BY (password rotation)',
       'ALTER USER ' || c_tmp_user || ' IDENTIFIED BY "Pf_' || DBMS_RANDOM.STRING('X', 12) || '2#"');

   ----------------------------------------------------------------------------------------
   -- 2. Granting system privileges the job does not own itself. Needs them WITH ADMIN
   --    OPTION, or GRANT ANY PRIVILEGE.
   ----------------------------------------------------------------------------------------
   try('GRANT CREATE SESSION, CREATE TABLE, CREATE VIEW',
       'GRANT CREATE SESSION, CREATE TABLE, CREATE VIEW TO ' || c_tmp_user);

   ----------------------------------------------------------------------------------------
   -- 3. Granting object privileges on objects owned by SYS / the DBMS_CLOUD owner. This is
   --    where a role-only user most often stops: it needs grant option on someone else's
   --    object, or GRANT ANY OBJECT PRIVILEGE.
   ----------------------------------------------------------------------------------------
   try('GRANT EXECUTE ON DBMS_CLOUD',
       'GRANT EXECUTE ON DBMS_CLOUD TO ' || c_tmp_user);
   try('GRANT READ, WRITE ON DIRECTORY DATA_PUMP_DIR',
       'GRANT READ, WRITE ON DIRECTORY DATA_PUMP_DIR TO ' || c_tmp_user);

   ----------------------------------------------------------------------------------------
   -- 4. Tablespace quota. DATA is the Autonomous tablespace name.
   ----------------------------------------------------------------------------------------
   try('ALTER USER ... QUOTA UNLIMITED ON DATA',
       'ALTER USER ' || c_tmp_user || ' QUOTA UNLIMITED ON DATA');

   ----------------------------------------------------------------------------------------
   -- 5. Network ACL. EXECUTE on a SYS-owned package; without it every external table read
   --    fails later with ORA-24247, long after provisioning appears to have worked.
   --    Exercised against a bogus host so nothing real is altered.
   ----------------------------------------------------------------------------------------
   BEGIN
      DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
         host => c_tmp_host,
         ace  => xs$ace_type(privilege_list => xs$name_list('connect'),
                             principal_name => c_tmp_user,
                             principal_type => xs_acl.ptype_db));
      report('DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE', TRUE);
   EXCEPTION
      WHEN OTHERS THEN report('DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE', FALSE, SQLERRM);
   END;

   ----------------------------------------------------------------------------------------
   -- 6. Registry, in THIS user's own schema - i.e. an unqualified registry_table. If this
   --    passes but you leave registry_table as ADMIN.EXT_REGISTRY_V4, the job still fails:
   --    writing into another schema needs the ANY TABLE privileges.
   ----------------------------------------------------------------------------------------
   quietly('DROP TABLE ' || c_tmp_tab || ' PURGE');
   try('CREATE TABLE in own schema (unqualified registry_table)',
       'CREATE TABLE ' || c_tmp_tab || ' (catalog_name VARCHAR2(128), owner VARCHAR2(128), '
       || 'table_name VARCHAR2(128), schema_fp VARCHAR2(64), synced_at TIMESTAMP, '
       || 'CONSTRAINT ' || c_tmp_tab || '_PK PRIMARY KEY(catalog_name, owner, table_name)) NOPARALLEL');
   try('MERGE into own registry',
       'MERGE INTO ' || c_tmp_tab || ' r USING (SELECT ''C'' catalog_name, ''O'' owner, '
       || '''T'' table_name, ''F'' schema_fp FROM dual) s '
       || 'ON (r.catalog_name=s.catalog_name AND r.owner=s.owner AND r.table_name=s.table_name) '
       || 'WHEN MATCHED THEN UPDATE SET r.schema_fp=s.schema_fp '
       || 'WHEN NOT MATCHED THEN INSERT(catalog_name,owner,table_name,schema_fp,synced_at) '
       || 'VALUES(s.catalog_name,s.owner,s.table_name,s.schema_fp,SYSTIMESTAMP)');

   ----------------------------------------------------------------------------------------
   -- 8. Visibility of other users - the job checks ALL_USERS before creating a schema.
   ----------------------------------------------------------------------------------------
   DECLARE n PLS_INTEGER;
   BEGIN
      SELECT COUNT(*) INTO n FROM all_users WHERE username = c_tmp_user;
      report('SELECT FROM ALL_USERS', n = 1,
             'the throwaway user is not visible in ALL_USERS');
   EXCEPTION
      WHEN OTHERS THEN report('SELECT FROM ALL_USERS', FALSE, SQLERRM);
   END;

   ----------------------------------------------------------------------------------------
   -- Cleanup
   ----------------------------------------------------------------------------------------
   BEGIN
      DBMS_NETWORK_ACL_ADMIN.REMOVE_HOST_ACE(
         host       => c_tmp_host,
         ace        => xs$ace_type(privilege_list => xs$name_list('connect'),
                                   principal_name => c_tmp_user,
                                   principal_type => xs_acl.ptype_db),
         remove_empty_acl => TRUE);
   EXCEPTION WHEN OTHERS THEN NULL;
   END;
   quietly('DROP TABLE ' || c_tmp_tab || ' PURGE');
   quietly('DROP USER '  || c_tmp_user || ' CASCADE');

   DBMS_OUTPUT.PUT_LINE(RPAD('-', 66, '-'));
   IF n_fail = 0 THEN
      DBMS_OUTPUT.PUT_LINE('RESULT: all checks passed - this user can drive the sync.');
      DBMS_OUTPUT.PUT_LINE('        Remember to set  registry_table: EXT_REGISTRY_V4  (unqualified)');
      DBMS_OUTPUT.PUT_LINE('        unless this user IS ADMIN.');
   ELSE
      DBMS_OUTPUT.PUT_LINE('RESULT: ' || n_fail || ' check(s) FAILED - see the errors above.');
      DBMS_OUTPUT.PUT_LINE('        Each failing line names the statement the job would issue;');
      DBMS_OUTPUT.PUT_LINE('        grant the missing privilege directly to this user and re-run.');
   END IF;
END;
/

-- Left behind on purpose if cleanup could not run. Verify nothing survived:
SELECT username FROM all_users  WHERE username  = 'ADW_SYNC_PREFLIGHT_TMP';
SELECT table_name FROM user_tables WHERE table_name = 'ADW_SYNC_PREFLIGHT_REG';
