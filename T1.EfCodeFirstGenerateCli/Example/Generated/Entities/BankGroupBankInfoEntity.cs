using System;

namespace Generated
{
    public class BankGroupBankInfoEntity
    {
        public int Id { get; set; }
        public int CustomerId { get; set; }
        public required string AccountId { get; set; }
        public required string CompanyName { get; set; }
        public required string BeneficiaryAddress { get; set; }
        public required string BankName { get; set; }
        public required string BankCode { get; set; }
        public required string BankAddress { get; set; }
        public required string BankSwiftCode { get; set; }
        public required string Currency { get; set; }
        public required string AccountNumber { get; set; }
        public required string CorrespondentBank { get; set; }
        public required string CorrespondentBankSwiftcode { get; set; }
        public required string IBAN { get; set; }
        public required string SortCode { get; set; }
        public required string RoutingCode { get; set; }
        public int Status { get; set; }
        public required string UpdatedBy { get; set; }
        public required string CreatedBy { get; set; }
        public DateTime? CreatedOn { get; set; }
        public DateTime? UpdatedOn { get; set; }
        public required string Jurisdiction { get; set; }
        public required string BankAccountType { get; set; }
        public required string BankBranch { get; set; }
        public required string SlipDetails { get; set; }
        public required string DisplayName { get; set; }
    }
}
