using System;

namespace Generated
{
    public class TimeZonesEntity
    {
        public int TimeZoneID { get; set; }
        public required string TimeZoneName { get; set; }
        public required string Presentation { get; set; }
        public decimal GMTOffSet { get; set; }
        public int Status { get; set; }
        public int CreatedBy { get; set; }
        public DateTime CreatedTime { get; set; }
        public int LastModifiedBy { get; set; }
        public DateTime LastModifiedTime { get; set; }
    }
}
