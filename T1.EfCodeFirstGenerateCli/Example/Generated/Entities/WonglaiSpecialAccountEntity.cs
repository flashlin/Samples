using System;

namespace Generated
{
    public class WonglaiSpecialAccountEntity
    {
        public int CustomerId { get; set; }
        public required string AccountId { get; set; }
        public DateTime? LastLoginDate { get; set; }
        public DateTime? TransactionCutOffDate { get; set; }
        public required string LastLoginIp { get; set; }
        public required string LastLoginCountry { get; set; }
        public DateTime? LastTransactionDate { get; set; }
        public int? OldExtraInfoId { get; set; }
    }
}
