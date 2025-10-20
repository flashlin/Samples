using System;

namespace Generated
{
    public class TableTrackerEntity
    {
        public int rid { get; set; }
        public required string tablename { get; set; }
        public long pk_id { get; set; }
        public DateTime tstamp { get; set; }
        public DateTime createdon { get; set; }
        public DateTime? LastModifiedOn { get; set; }
    }
}
