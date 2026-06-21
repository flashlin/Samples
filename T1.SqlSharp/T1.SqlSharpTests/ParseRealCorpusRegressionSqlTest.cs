using FluentAssertions;
using T1.SqlSharp.ParserLit;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseRealCorpusRegressionSqlTest
{
    [TestCaseSource(nameof(SqlCases))]
    public void Parse_real_corpus_error_case(string sourceFile, string sql)
    {
        var parser = new SqlParser(sql);
        var results = parser.ExtractStatementResults().ToList();
        results.Should().NotBeEmpty(sourceFile);
        results.Should().OnlyContain(x => !x.HasError, sourceFile);
        results.Should().OnlyContain(x => x.Result != null, sourceFile);
    }

    private static IEnumerable<TestCaseData> SqlCases()
    {
        yield return Case(
            "AccountNotificationAPI_DeleteOldTelegramNotificationLog_21.10.sql",
            """
            CREATE PROCEDURE [dbo].[AccountNotificationAPI_DeleteOldTelegramNotificationLog_21.10]
                @cutOffDate DATE,
                @isUat BIT
            AS
            BEGIN
                SET NOCOUNT ON;
                SET DEADLOCK_PRIORITY low;
                SELECT 1
                WHILE (@@ROWCOUNT <> 0)
                BEGIN
                    DELETE TOP (4000)
                    FROM [dbo].[TelegramNotificationLog]
                    WHERE CreatedOn <= @cutOffDate
                        AND IsUat = @isUat
                END
            END
            """);

        yield return Case(
            "Sch_Report_OnlineUserSB.sql",
            """
            CREATE PROCEDURE [dbo].[Sch_Report_OnlineUserSB]
            AS
            BEGIN
                DECLARE @asiaTotal AS INT = 0;
                DECLARE @asiaTotalQ AS INT = 0;
                DECLARE @euroTotal AS INT = 0;
                SELECT @asiaTotal = @asiaTotal + @asiaTotalQ
            END
            """);

        yield return Case(
            "m9_dotnet_csmaTotalByMemberx.sql",
            """
            CREATE PROCEDURE [dbo].[m9_dotnet_csmaTotalByMemberx]
                @agentid AS INT,
                @fdate AS DATETIME,
                @tdate AS DATETIME
            AS
            SELECT c.username, m.*
            FROM (
                SELECT custid, SUM(turnover) turnover
                FROM DailyStatement
                GROUP BY custid
            ) m
            INNER JOIN customer c WITH (NOLOCK) ON c.custid = m.custid
            """);

        yield return Case(
            "Admin_SB_Pluto_GetRisk1x2TS_6.6.sql",
            """
            CREATE PROCEDURE [dbo].[Admin_SB_Pluto_GetRisk1x2TS_6.6]
                @tStamp AS TIMESTAMP
            AS
            BEGIN
                SET NOCOUNT ON
                IF @tStamp > 0
                BEGIN
                    SELECT R.OddsID, R.MatchID, R.TStamp
                    FROM Risk1x2 R WITH (NOLOCK)
                    WHERE R.TStamp > @tStamp
                END
            END
            """);

        yield return Case(
            "Admin_SB_EventMgmt_AddOddsForNewEvent_1.0.1.sql",
            """
            CREATE PROCEDURE [dbo].[Admin_SB_EventMgmt_AddOddsForNewEvent_1.0.1]
            AS
            BEGIN
                DECLARE @dbError AS INT = 0
                DECLARE @errorCode AS INT = 0
                DECLARE @oddsID AS INT = 0
                SELECT @dbError = @@error
                IF @dbError <> 0
                BEGIN
                    SELECT @errorCode AS ErrorCode
                END
                SELECT @oddsID = SCOPE_IDENTITY()
            END
            """);

        yield return Case(
            "Pontus_SB_GetStartedTransaction_4.3.sql",
            """
            CREATE PROCEDURE [dbo].[Pontus_SB_GetStartedTransaction_4.3]
                @timeStamp AS DATETIME
            AS
            BEGIN
                IF EXISTS (SELECT 1 FROM TransactionInfo WITH (NOLOCK) WHERE ModifiedOn > @timeStamp)
                BEGIN
                    SELECT TOP 1 [no], [status]
                    FROM TransactionInfo WITH (NOLOCK)
                    WHERE ModifiedOn > @timeStamp
                    ORDER BY [no] DESC
                END
            END
            """);

        yield return Case(
            "PlutoReplication_RB_TruncateBuffer_RacingBet_14.05.sql",
            """
            CREATE PROCEDURE [dbo].[PlutoReplication_RB_TruncateBuffer_RacingBet_14.05]
            WITH EXECUTE AS 'plutoproxy'
            AS
            BEGIN
                SET NOCOUNT ON;
                TRUNCATE TABLE RacingBet_buffertable
            END
            """);

        yield return Case(
            "AccountAPI_InsertSportsRiskControl_20.05.sql",
            """
            CREATE PROCEDURE [dbo].[AccountAPI_InsertSportsRiskControl_20.05]
                @tvpSetting [dbo].[SportsRiskControlSetting] READONLY
            AS
            BEGIN
                INSERT INTO [dbo].[SportsRiskControl]
                (
                    [CustomerId],
                    [CreatedOn],
                    [ModifiedBy]
                )
                SELECT [CustomerId], GETDATE(), [ModifiedBy]
                FROM @tvpSetting
            END
            """);

        yield return Case(
            "BetGenius_DeleteFixture_1.0.0.sql",
            """
            CREATE PROCEDURE [dbo].[BetGenius_DeleteFixture_1.0.0]
                @fixtureId AS INT
            AS
            BEGIN
                IF @fixtureId != 18784
                BEGIN
                    DELETE FROM Fixture WHERE FixtureId = @fixtureId
                END
            END
            """);

        yield return Case(
            "Aither_LC_Player_VerifyUser_14.02.sql",
            """
            CREATE PROCEDURE [dbo].[Aither_LC_Player_VerifyUser_14.02]
                @username AS NVARCHAR(50),
                @retcode AS INT OUTPUT
            AS
            BEGIN
                SELECT @retcode = 0
                RETURN @retcode
            END
            """);

        yield return Case(
            "GamesBetAll.sql",
            """
            CREATE SYNONYM [dbo].[GamesBetAll]
            FOR [REMOTEREP13].[PlutoRepGM].[dbo].[GamesBet]
            """);

        yield return Case(
            "stmt201201.sql",
            """
            ALTER DATABASE [$(DatabaseName)]
            ADD FILEGROUP [stmt201201]
            """);

        yield return Case(
            "Logins.sql",
            """
            CREATE LOGIN [L_Earth]
            WITH PASSWORD = 'titan@2006',
                CHECK_POLICY = OFF
            """);

        yield return Case(
            "Local.PreDeployment.sql",
            """
            :r ..\..\CommonFiles\Scripts\CommonPreDeployment.sql
            """);

        yield return Case(
            "AliasTag.sql",
            """
            CREATE NONCLUSTERED INDEX [IX_AliasTag_TagAliasId]
            ON [dbo].[AliasTag] ([TagAliasId] ASC)
            WITH (PAD_INDEX = OFF, ONLINE = ON, FILLFACTOR = 80)
            ON [PRIMARY]
            """);

        yield return Case(
            "Leo_Transfer_SingleTransfer_14.02.sql",
            """
            CREATE PROCEDURE [dbo].[Leo_Transfer_SingleTransfer_14.02]
            AS
            BEGIN
                DECLARE @ybal AS DECIMAL(19, 6)
                DECLARE @amt AS DECIMAL(19, 6) = 1
                DECLARE @adjustedAmt AS DECIMAL(19, 6)
                EXEC [dbo].[Leo_Account_GetYesterdayTotalBalance_5.4] @fcustid, @froleid, @ybal OUTPUT
                IF ABS(@ybal + @amt) < 0.01
                BEGIN
                    SET @adjustedAmt = -(@ybal)
                END
            END
            """);

        yield return Case(
            "DisplayNamePrefix.sql",
            """
            EXEC sp_addextendedproperty
                @name = N'MS_Description',
                @value = N'Identifier',
                @level0type = N'SCHEMA',
                @level0name = N'dbo',
                @level1type = N'TABLE',
                @level1name = N'DisplayNamePrefix',
                @level2type = N'COLUMN',
                @level2name = N'Id'
            """);

        yield return Case(
            "DeleteGroup.sql",
            """
            CREATE PROCEDURE [dbo].[DeleteGroup]
            AS
            BEGIN
                DECLARE @cre_by AS NVARCHAR(50) = N'root'
                DECLARE @cre_on AS DATETIME = GETDATE()
                INSERT INTO [dbo].[AuditLog]
                VALUES ('ALL', '0', 'ALL', 'ALL', 'DELETE', @cre_by, @cre_on)
            END
            """);

        yield return Case(
            "DailyStatement.sql",
            """
            GO
            GO
            """);

        yield return Case(
            "Report_Status.sql",
            """
            EXEC sys.sp_addextendedproperty
                @name = N'MS_Description',
                @value = N'Report date',
                @level0type = N'SCHEMA',
                @level0name = N'dbo',
                @level1type = N'TABLE',
                @level1name = N'Report_Status',
                @level2type = N'COLUMN',
                @level2name = N'Date'
            """);

        yield return Case(
            "User.sql",
            """
            CREATE USER [L_Earth]
            FOR LOGIN [L_Earth]
            WITH DEFAULT_SCHEMA = [dbo]
            """);

        yield return Case(
            "Enum.sql",
            """
            INSERT [dbo].[Enum] ([Name], [Option])
            VALUES (N'AgentsTransferSetting', N'Friday')
            """);

        yield return Case(
            "Account_Upsert_TransferDailyStatement_20.07.sql",
            """
            WAITFOR DELAY '00:00:00:003'
            """);

        yield return Case(
            "adm_check_dbspace.sql",
            """
            DBCC UPDATEUSAGE(0)
            """);

        yield return Case(
            "Admin_SB_Settlement_Settle_Early_Sure_Win_Bets_1.1.0.sql",
            """
            CREATE TABLE #tempBetTrans
            (
                TransID bigint,
                MatchResultId int
            )
            """);

        yield return Case(
            "adm_sch_move_historylog_1.0.0.sql",
            """
            ALTER PARTITION FUNCTION pf14Log1()
            SPLIT RANGE (@nextdate)
            """);

        yield return Case(
            "psProductType.sql",
            """
            CREATE PARTITION SCHEME [psProductType]
            AS PARTITION [pfProductType]
            TO ([PRIMARY], [PRIMARY], [PRIMARY])
            """);

        yield return Case(
            "adm_sch_AutoClaim.sql",
            """
            SET @getData = CURSOR FOR
            SELECT ID, CustID
            FROM JoinNowPromotion WITH (NOLOCK)
            """);

        yield return Case(
            "AccountGamesBetAPI_Merge_PlayerStatement_22.02.sql",
            """
            MERGE INTO PlayerStatement AS target
            USING (SELECT @customerId, @playerWinLoss) AS source (custId, playerWinLoss)
            ON target.custId = source.custId
            WHEN MATCHED THEN
                UPDATE SET playerWinLoss = source.playerWinLoss
            WHEN NOT MATCHED THEN
                INSERT (custId, playerWinLoss)
                VALUES (source.custId, source.playerWinLoss)
            """);

        yield return Case(
            "Admin_SB_Settlement_Unvoid_UnSettled_14.08.sql",
            """
            UPDATE b
            SET Ruben = 0,
                Status = 'running'
            OUTPUT inserted.transid, inserted.custid
            FROM bettrans b
            WHERE b.transid = @transid
            """);
    }

    private static TestCaseData Case(string sourceFile, string sql)
    {
        return new TestCaseData(sourceFile, sql).SetName(sourceFile);
    }
}
