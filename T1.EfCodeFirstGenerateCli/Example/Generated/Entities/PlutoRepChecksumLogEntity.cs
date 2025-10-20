using System;

namespace Generated
{
    public class PlutoRepChecksumLogEntity
    {
        public required string TableName { get; set; }
        public int Count { get; set; }
        public required string Remark { get; set; }
        public int? SourceCount { get; set; }
        public decimal? SourceSum { get; set; }
        public int? RepCount { get; set; }
        public decimal? RepSum { get; set; }
        public required byte[] TStampMin { get; set; }
        public required byte[] TStampMax { get; set; }
        public DateTime? MTimeMin { get; set; }
        public DateTime? MTimeMax { get; set; }
        public DateTime CreatedOn { get; set; }
        public required string CreatedBy { get; set; }
    }
}
