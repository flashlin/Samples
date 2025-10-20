using System;

namespace Generated
{
    public class UnsettledOrdersEntity
    {
        public long TransactionId { get; set; }
        public int CustomerId { get; set; }
        public DateTime TransactionDate { get; set; }
        public int PlayerStatus { get; set; }
        public int CurrencyId { get; set; }
        public int RoleId { get; set; }
        public decimal Stake { get; set; }
        public decimal ActualStake { get; set; }
        public decimal CommissionRate { get; set; }
        public decimal PositionTaking { get; set; }
    }
}
