using System;

namespace Generated
{
    public class AgentsBetSettingEntity
    {
        public int custid { get; set; }
        public int sportid { get; set; }
        public int bettype { get; set; }
        public int? parentid { get; set; }
        public decimal? minbet { get; set; }
        public decimal? maxbet { get; set; }
        public decimal? maxpermatch { get; set; }
        public required string remark { get; set; }
        public required string modifiedBy { get; set; }
        public DateTime? modifiedDate { get; set; }
        public decimal? credit { get; set; }
        public decimal? maxcredit { get; set; }
        public DateTime? lastTxnDate { get; set; }
    }
}
