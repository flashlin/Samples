using System;

namespace Generated
{
    public class RiskControlBalanceUpsertsEntity
    {
        public int Id { get; set; }
        public int CustomerId { get; set; }
        public decimal WinLose { get; set; }
        public decimal StartingRiskBalance { get; set; }
        public bool? TryReset { get; set; }
        public DateTime RequestTime { get; set; }
        public bool IsProcessed { get; set; }
    }
}
