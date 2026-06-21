using FluentAssertions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseRealCorpusRegressionSqlTest
{
    [TestCaseSource(nameof(SqlCases))]
    public void Parse_real_corpus_error_case(string sourceFile, string sql)
    {
        var rc = sql.ParseSql();
        rc.HasError.Should().BeFalse($"{sourceFile}: {rc.Error.Message}");
        rc.Result.Should().NotBeNull(sourceFile);
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
    }

    private static TestCaseData Case(string sourceFile, string sql)
    {
        return new TestCaseData(sourceFile, sql).SetName(sourceFile);
    }
}
