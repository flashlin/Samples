using System;

namespace Generated
{
    public class BonusWalletEntity
    {
        public long BonusWalletId { get; set; }
        public int CustomerId { get; set; }
        public int ProductType { get; set; }
        public required string GroupKey { get; set; }
        public byte Status { get; set; }
        public DateTime CreatedOn { get; set; }
        public DateTime? LastModifiedOn { get; set; }
    }
}
