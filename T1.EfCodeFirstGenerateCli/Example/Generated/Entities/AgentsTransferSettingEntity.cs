using System;

namespace Generated
{
    public class AgentsTransferSettingEntity
    {
        public int CustID { get; set; }
        public required string UserName { get; set; }
        public int TransferSetting { get; set; }
        public int ParentID { get; set; }
        public required string CreatedBy { get; set; }
        public DateTime CreatedOn { get; set; }
        public required string ModifiedBy { get; set; }
        public DateTime ModifiedOn { get; set; }
    }
}
