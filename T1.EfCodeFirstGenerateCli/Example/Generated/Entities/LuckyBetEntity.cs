using System;

namespace Generated
{
    public class LuckyBetEntity
    {
        public int CustId { get; set; }
        public DateTime? IdleTime { get; set; }
        public DateTime? ActiveTime { get; set; }
        public DateTime? ButtonClickTime { get; set; }
        public required string HostId { get; set; }
    }
}
