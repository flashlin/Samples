using System;

namespace Generated
{
    public class MoneyTransferBankInfoEntity
    {
        public int MTBID { get; set; }
        public int? CustId { get; set; }
        public required string ISOCUrrency { get; set; }
        public required string AccountId { get; set; }
        public required string AccountHolderName { get; set; }
        public required string AccountNumber { get; set; }
        public required string BankName { get; set; }
        public required string Branch { get; set; }
        public required string BankAddress { get; set; }
        public required string IBAN { get; set; }
        public required string SWIFT { get; set; }
        public required string SortCode { get; set; }
        public DateTime? LastUsedOn { get; set; }
        public required string BeneficiaryAddress { get; set; }
    }
}
