using System;

namespace Generated
{
    public class MemberBetSettingEntity
    {
        public int custid { get; set; }
        public int sportid { get; set; }
        public int bettype { get; set; }
        public int recommend { get; set; }
        public decimal? minbet { get; set; }
        public decimal? maxbet { get; set; }
        public decimal? maxpermatch { get; set; }
        public required string remark { get; set; }
        public required string modifiedBy { get; set; }
        public DateTime? modifiedDate { get; set; }
        public decimal? credit { get; set; }
        public DateTime? lastBetDate { get; set; }
        public DateTime? lastTxnDate { get; set; }
        public DateTime? lastWinLostDate { get; set; }
    }
}
