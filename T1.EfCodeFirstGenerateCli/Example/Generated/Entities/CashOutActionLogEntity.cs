using System;

namespace Generated
{
    public class CashOutActionLogEntity
    {
        public long LogId { get; set; }
        public long transid { get; set; }
        public int custid { get; set; }
        public DateTime transdate { get; set; }
        public required string status { get; set; }
        public decimal? winlost { get; set; }
        public decimal? awinlost { get; set; }
        public decimal? mwinlost { get; set; }
        public decimal? swinlost { get; set; }
        public decimal? playercomm { get; set; }
        public decimal? comm { get; set; }
        public decimal? acomm { get; set; }
        public decimal? scomm { get; set; }
        public DateTime? winlostdate { get; set; }
        public byte? statuswinlost { get; set; }
        public int? betstatus { get; set; }
        public int? MemberStatus { get; set; }
        public DateTime? LastModifiedOn { get; set; }
        public DateTime? CashOutTime { get; set; }
        public decimal? CashOutValue { get; set; }
        public int? ActionType { get; set; }
        public DateTime? LogDate { get; set; }
    }
}
