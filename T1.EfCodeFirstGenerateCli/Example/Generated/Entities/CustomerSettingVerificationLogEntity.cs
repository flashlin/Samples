using System;

namespace Generated
{
    public class CustomerSettingVerificationLogEntity
    {
        public int CustId { get; set; }
        public required string Username { get; set; }
        public short RoleId { get; set; }
        public short ProductType { get; set; }
        public required string TableName { get; set; }
        public DateTime CustomerCreatedDate { get; set; }
        public DateTime? CreatedOn { get; set; }
        public bool HasTransaction { get; set; }
    }
}
