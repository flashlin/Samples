using System;

namespace Generated
{
    public class SettledOrdersEntity
    {
        public long TransactionId { get; set; }
        public int CustomerId { get; set; }
        public int PlayerStatus { get; set; }
        public int CurrencyId { get; set; }
        public int RoleId { get; set; }
        public decimal Stake { get; set; }
        public decimal ActualStake { get; set; }
        public decimal CommissionableStake { get; set; }
        public decimal TurnoverStake { get; set; }
        public decimal WinLost { get; set; }
        public decimal CommissionRate { get; set; }
        public decimal Commission { get; set; }
        public decimal PositionTaking { get; set; }
        public DateTime WinlostDate { get; set; }
        public int TransactionStatus { get; set; }
        public byte IsResettled { get; set; }
        public DateTime CreatedOn { get; set; }
        public long Id { get; set; }
    }
}
