using System;

namespace Generated
{
    public class PTDirectPresetEntity
    {
        public int CustomerId { get; set; }
        public int ParentId { get; set; }
        public decimal ParentMinimum { get; set; }
        public decimal ParentForce { get; set; }
        public required string LastModifiedBy { get; set; }
        public DateTime? LastModifiedDate { get; set; }
    }
}
