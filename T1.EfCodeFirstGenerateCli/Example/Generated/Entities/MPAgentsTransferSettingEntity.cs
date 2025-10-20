using System;

namespace Generated
{
    public class MPAgentsTransferSettingEntity
    {
        public int CustomerId { get; set; }
        public required string UserName { get; set; }
        public int TransferSetting { get; set; }
        public int ParentId { get; set; }
        public required string CreatedBy { get; set; }
        public DateTime CreatedOn { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime ModifiedOn { get; set; }
    }
}
