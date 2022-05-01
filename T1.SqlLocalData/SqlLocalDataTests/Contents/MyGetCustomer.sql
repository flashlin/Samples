CREATE PROC MyGetCustomer 
	@id INT AS 
BEGIN 
	SET NOCOUNT ON; 
	select name from customer 
	WHERE id=@id 
END
