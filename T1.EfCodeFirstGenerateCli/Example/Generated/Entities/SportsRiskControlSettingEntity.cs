using System;

namespace Generated
{
    public class SportsRiskControlSettingEntity
    {
        public int CustomerId { get; set; }
        public long MaxWin { get; set; }
        public long MaxLose { get; set; }
        public bool IsMaxWinUnlimited { get; set; }
        public bool IsMaxLoseUnlimited { get; set; }
        public int ResetProfileId { get; set; }
        public DateTime ModifiedOn { get; set; }
        public required string ModifiedBy { get; set; }
    }
}
