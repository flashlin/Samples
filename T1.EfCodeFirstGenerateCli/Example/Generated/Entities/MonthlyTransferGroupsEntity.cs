using System;

namespace Generated
{
    public class MonthlyTransferGroupsEntity
    {
        public int Id { get; set; }
        public int GroupId { get; set; }
        public required string GroupName { get; set; }
        public DateTime CreatedOn { get; set; }
        public required string CreatedBy { get; set; }
        public DateTime? ModifiedOn { get; set; }
        public required string ModifiedBy { get; set; }
        public bool IsDeleted { get; set; }
    }
}
