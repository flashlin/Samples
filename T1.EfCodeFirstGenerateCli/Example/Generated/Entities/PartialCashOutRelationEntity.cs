using System;

namespace Generated
{
    public class PartialCashOutRelationEntity
    {
        public long Id { get; set; }
        public long TransId { get; set; }
        public long OriginalTransId { get; set; }
        public DateTime CreatedOn { get; set; }
        public DateTime? LastModifiedOn { get; set; }
        public bool IsDeleted { get; set; }
    }
}
