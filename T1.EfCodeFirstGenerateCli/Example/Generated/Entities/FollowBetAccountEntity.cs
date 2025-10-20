using System;

namespace Generated
{
    public class FollowBetAccountEntity
    {
        public int CustID { get; set; }
        public required string AccountID { get; set; }
        public required string ISOCurrency { get; set; }
        public int FollowCustID { get; set; }
        public required string FollowAccountID { get; set; }
        public bool IsEnabled { get; set; }
        public decimal FollowPercentage { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime? ModifiedOn { get; set; }
    }
}
