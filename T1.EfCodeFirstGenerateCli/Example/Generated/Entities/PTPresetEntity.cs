using System;

namespace Generated
{
    public class PTPresetEntity
    {
        public int custid { get; set; }
        public byte roleid { get; set; }
        public int parentid { get; set; }
        public byte type { get; set; }
        public int typeid { get; set; }
        public byte? priority { get; set; }
        public decimal _5050mktPTMin { get; set; }
        public decimal _5050mktPTMax { get; set; }
        public byte _5050mktTakeAll { get; set; }
        public decimal non5050mktPTMin { get; set; }
        public decimal non5050mktPTMax { get; set; }
        public byte non5050mktTakeAll { get; set; }
        public decimal HDPLivemktPTMin { get; set; }
        public decimal HDPLivemktPTMax { get; set; }
        public byte HDPLivemktTakeAll { get; set; }
        public decimal OULivemktPTMin { get; set; }
        public decimal OULivemktPTMax { get; set; }
        public byte OULivemktTakeAll { get; set; }
        public decimal ORmktPTMin { get; set; }
        public decimal ORmktPTMax { get; set; }
        public byte ORmktTakeAll { get; set; }
        public DateTime? LastModifiedDate { get; set; }
    }
}
