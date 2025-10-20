using System;

namespace Generated
{
    public class TableTrackerBetSettingEntity
    {
        public int rid { get; set; }
        public int CustId { get; set; }
        public int SportId { get; set; }
        public int BetType { get; set; }
        public DateTime ModifiedOn { get; set; }
        public required string TableName { get; set; }
    }
}
