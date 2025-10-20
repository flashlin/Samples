using System;

namespace Generated
{
    public class MpPlayerCommissionEntity
    {
        public int CustomerId { get; set; }
        public int ParentId { get; set; }
        public int Type { get; set; }
        public decimal Commission { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime ModifiedDate { get; set; }
    }
}
