using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseBackupRestoreKillSqlTest
{
    [Test]
    public void Kill_session_id()
    {
        var sql = "KILL 55";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlKillStatement
        {
            SessionId = "55"
        });
    }

    [Test]
    public void Kill_with_statusonly()
    {
        var sql = "KILL 55 WITH STATUSONLY";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlKillStatement
        {
            SessionId = "55",
            WithStatusOnly = true
        });
    }

    [Test]
    public void Backup_database_to_disk()
    {
        var sql = "BACKUP DATABASE Sales TO DISK = 'C:\\sales.bak'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlBackupRestoreStatement
        {
            IsBackup = true,
            ObjectKind = "DATABASE",
            Name = "Sales",
            Devices = ["DISK = 'C:\\sales.bak'"]
        });
    }

    [Test]
    public void Backup_database_with_options()
    {
        var sql = "BACKUP DATABASE Sales TO DISK = 'C:\\sales.bak' WITH INIT, STATS = 10";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlBackupRestoreStatement
        {
            IsBackup = true,
            ObjectKind = "DATABASE",
            Name = "Sales",
            Devices = ["DISK = 'C:\\sales.bak'"],
            Options = ["INIT", "STATS = 10"]
        });
    }

    [Test]
    public void Restore_database_from_disk()
    {
        var sql = "RESTORE DATABASE Sales FROM DISK = 'C:\\sales.bak' WITH REPLACE";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlBackupRestoreStatement
        {
            IsBackup = false,
            ObjectKind = "DATABASE",
            Name = "Sales",
            Devices = ["DISK = 'C:\\sales.bak'"],
            Options = ["REPLACE"]
        });
    }

    [Test]
    public void Backup_log_to_disk()
    {
        var sql = "BACKUP LOG Sales TO DISK = 'C:\\sales.trn'";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlBackupRestoreStatement
        {
            IsBackup = true,
            ObjectKind = "LOG",
            Name = "Sales",
            Devices = ["DISK = 'C:\\sales.trn'"]
        });
    }
}
