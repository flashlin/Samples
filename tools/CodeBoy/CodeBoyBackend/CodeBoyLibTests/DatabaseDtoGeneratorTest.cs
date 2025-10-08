using CodeBoyLib.Services;
using FluentAssertions;
using NUnit.Framework;

namespace CodeBoyLibTests
{
    [TestFixture]
    public class DatabaseDtoGeneratorTest
    {
        [Test]
        public void GenerateEfDtoCode_WithAutoWithdrawBalanceTable_ShouldGenerateCorrectDto()
        {
            var createTableSql = @"CREATE TABLE [dbo].[AutoWithdrawBalance]
(
[CustomerId] [int] NOT NULL,
[BalanceType] int NOT NULL,
[Amount] decimal(19,6) NOT NULL CONSTRAINT [DF_AutoWithdrawBalance_Amount] default(0),
[CheckTime] DATETIME NOT NULL,
[BetCheckTime] DATETIME NOT NULL DEFAULT GETDATE(),
[CreatedOn] DATETIME NOT NULL CONSTRAINT [DF_AutoWithdrawBalance_CreatedOn] DEFAULT GETDATE(), 
[CreatedBy] NVARCHAR(50) NOT NULL,
[ModifiedOn] DATETIME NOT NULL CONSTRAINT [DF_AutoWithdrawBalance_ModifiedOn] DEFAULT GETDATE(), 
[ModifiedBy] NVARCHAR(50) NULL,
[ResetOn] DATETIME NULL,
[ResetBy] NVARCHAR(50) NULL
)";

            var expected = @"public class AutoWithdrawBalanceDto {
	public int CustomerId { get; set; }
	public int BalanceType { get; set; }
	public decimal Amount { get; set; }
	public DateTime CheckTime { get; set; }
	public DateTime BetCheckTime { get; set; }
	public DateTime CreatedOn { get; set; }
	public string CreatedBy { get; set; }
	public DateTime ModifiedOn { get; set; }
	public string ModifiedBy { get; set; }
	public DateTime ResetOn { get; set; }
	public string ResetBy { get; set; }
}
";

            var generator = new DatabaseDtoGenerator();
            var result = generator.GenerateEfDtoCode(createTableSql);

            result.Should().Be(expected);
        }
    }
}

