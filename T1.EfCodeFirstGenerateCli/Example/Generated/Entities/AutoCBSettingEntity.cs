using System;

namespace Generated
{
    public class AutoCBSettingEntity
    {
        public int ID { get; set; }
        public int? ProductType { get; set; }
        public int? SubProductType { get; set; }
        public bool? isIOM { get; set; }
        public int? FromID { get; set; }
        public required string FromAccount { get; set; }
        public int? ToID { get; set; }
        public required string ToAccount { get; set; }
        public int? TxnType { get; set; }
        public required string Description { get; set; }
        public bool? IsTest { get; set; }
        public int? AutoCBType { get; set; }
    }
}
