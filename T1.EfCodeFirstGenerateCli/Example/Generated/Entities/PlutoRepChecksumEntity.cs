using System;

namespace Generated
{
    public class PlutoRepChecksumEntity
    {
        public int Id { get; set; }
        public required string TableName { get; set; }
        public required byte[] LastSyncTimeStamp { get; set; }
        public DateTime? LastSyncDateTime { get; set; }
        public long? LastSyncId { get; set; }
        public int? LastSyncCount { get; set; }
        public DateTime? LastRepExeTime { get; set; }
        public required byte[] LastChecksumTimeStampMin { get; set; }
        public DateTime? LastChecksumDateTimeMin { get; set; }
        public long? LastChecksumIdMin { get; set; }
        public required byte[] LastChecksumTimeStampMax { get; set; }
        public DateTime? LastChecksumDateTimeMax { get; set; }
        public long? LastChecksumIdMax { get; set; }
        public int? LastChecksumCountMain { get; set; }
        public decimal? LastChecksumValueMain { get; set; }
        public int? LastChecksumCountRep { get; set; }
        public decimal? LastChecksumValueRep { get; set; }
        public DateTime? LastChecksumExeTime { get; set; }
        public required byte[] NextChecksumTimeStamp { get; set; }
        public DateTime? NextChecksumDatetime { get; set; }
        public long? NextChecksumId { get; set; }
        public required byte[] LastAdminCheckSumTS { get; set; }
        public DateTime? LastAdminCheckSumTime { get; set; }
        public long? LastAdminCheckSumId { get; set; }
    }
}
