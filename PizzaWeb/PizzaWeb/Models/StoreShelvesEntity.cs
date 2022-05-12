using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace PizzaWeb.Models
{
	[Table("StoreShelves")]
	public class StoreShelvesEntity
	{
		[Key]
		public int Id { get; set; }
		public string Title { get; set; }
		public string Content { get; set; }
		public string ImageName { get; set; }
	}
}
