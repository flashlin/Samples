using System;

namespace Generated
{
    public class SettleTypeDisplayTypeRelationEntity
    {
        public int SettleTypeID { get; set; }
        public int DisplayTypeID { get; set; }
        public bool IsFirstHalf { get; set; }
        public DateTime LastModifiedOn { get; set; }
        public required string ModifiedBy { get; set; }
        public bool? Is5050 { get; set; }
    }
}
