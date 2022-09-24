// See https://aka.ms/new-console-template for more information

using var context = new DemoDbContext();
DemoDbContext.Initialize(context);


var q1 = from tb1 in context.Customers
			select tb1;
context.WriteLog(() =>
{
	var customer = q1.ToList();
});



var q2 = from tb1 in context.Customers
			join tb2 in context.Houses on tb1.Id equals tb2.CustomerId
			select new
			{
				tb1.Name,
				tb2.Address
			};
context.WriteLog(() =>
{
	q2.ToList();
});


Console.WriteLine("Hello, World!");