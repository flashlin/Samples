using System;

namespace Generated
{
    public class ChangeParentLogEntity
    {
        public int LogID { get; set; }
        public required string UserName { get; set; }
        public required string OldParentName { get; set; }
        public required string NewParentName { get; set; }
        public DateTime? ModifiedDate { get; set; }
        public required string ModifiedBy { get; set; }
    }
}
