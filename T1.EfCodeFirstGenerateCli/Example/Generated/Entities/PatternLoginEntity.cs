using System;

namespace Generated
{
    public class PatternLoginEntity
    {
        public int CustID { get; set; }
        public required string Pattern { get; set; }
        public int Status { get; set; }
        public int FailCount { get; set; }
        public DateTime CreatedOn { get; set; }
        public required string CreatedBy { get; set; }
        public DateTime ModifiedOn { get; set; }
        public required string ModifiedBy { get; set; }
    }
}
