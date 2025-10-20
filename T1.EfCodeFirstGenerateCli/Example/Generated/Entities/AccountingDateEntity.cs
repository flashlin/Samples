using System;

namespace Generated
{
    public class AccountingDateEntity
    {
        public required string ActionType { get; set; }
        public byte ProductType { get; set; }
        public DateTime? LastActionDate { get; set; }
        public DateTime ModifiedOn { get; set; }
        public required string ModifiedBy { get; set; }
    }
}
