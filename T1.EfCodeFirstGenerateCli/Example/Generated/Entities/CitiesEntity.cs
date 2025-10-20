using System;

namespace Generated
{
    public class CitiesEntity
    {
        public int CityID { get; set; }
        public required string CityName { get; set; }
        public int Status { get; set; }
        public int CreatedBy { get; set; }
        public DateTime CreatedTime { get; set; }
        public int LastModifiedBy { get; set; }
        public DateTime LastModifiedTime { get; set; }
        public int CountryID { get; set; }
    }
}
