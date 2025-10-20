using System;

namespace Generated
{
    public class BlackListWordEntity
    {
        public int BlackListWordID { get; set; }
        public required string BlackListWord { get; set; }
        public int BlackListWordType { get; set; }
        public byte Status { get; set; }
        public DateTime LastModifiedOn { get; set; }
        public required string LastModifiedBy { get; set; }
        public required string CreatedBy { get; set; }
    }
}
