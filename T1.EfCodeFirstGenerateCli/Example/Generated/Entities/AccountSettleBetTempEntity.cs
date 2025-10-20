using System;

namespace Generated
{
    public class AccountSettleBetTempEntity
    {
        public int BatchId { get; set; }
        public long TransId { get; set; }
        public decimal winlost { get; set; }
        public DateTime winlostdate { get; set; }
        public required string status { get; set; }
        public byte statuswinlost { get; set; }
        public int betstatus { get; set; }
        public byte ruben { get; set; }
        public decimal CommissionableStake { get; set; }
        public bool IsProcessed { get; set; }
    }
}
