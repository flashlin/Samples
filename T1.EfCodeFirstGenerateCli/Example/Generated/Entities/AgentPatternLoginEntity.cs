using System;

namespace Generated
{
    public class AgentPatternLoginEntity
    {
        public int CustomerId { get; set; }
        public int RoleId { get; set; }
        public required string PatternPassword { get; set; }
        public DateTime? CreatedOn { get; set; }
        public required string CreatedBy { get; set; }
    }
}
