using System;

namespace Generated
{
    public class AgentsMaxPTEntity
    {
        public int CustID { get; set; }
        public required string UserName { get; set; }
        public byte RoleID { get; set; }
        public decimal MaxPT { get; set; }
        public required string Remark { get; set; }
        public DateTime CreatedOn { get; set; }
        public required string CreatedBy { get; set; }
    }
}
