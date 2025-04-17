-- Create Database
CREATE DATABASE TestDatabase;
GO

USE TestDatabase;
GO

-- Create Login and Role
CREATE LOGIN TestUser WITH PASSWORD = 'Test@123456';
CREATE USER TestUser FOR LOGIN TestUser;
GO

CREATE ROLE TestRole;
GO

ALTER ROLE TestRole ADD MEMBER TestUser;
GO

-- Create Tables with Relationships
CREATE TABLE Categories (
    CategoryId INT IDENTITY(1,1) PRIMARY KEY,
    CategoryName NVARCHAR(50) NOT NULL,
    CreatedDate DATETIME DEFAULT GETDATE()
);

CREATE TABLE Products (
    ProductId INT IDENTITY(1,1) PRIMARY KEY,
    CategoryId INT NOT NULL,
    ProductName NVARCHAR(100) NOT NULL,
    UnitPrice DECIMAL(18,2) DEFAULT 0.00,
    CreatedDate DATETIME DEFAULT GETDATE(),
    CONSTRAINT FK_Products_Categories FOREIGN KEY (CategoryId) 
        REFERENCES Categories(CategoryId)
);

CREATE TABLE OrderDetails (
    OrderDetailId INT IDENTITY(1,1) PRIMARY KEY,
    ProductId INT NOT NULL,
    Quantity INT NOT NULL DEFAULT 1,
    OrderDate DATETIME DEFAULT GETDATE(),
    CONSTRAINT FK_OrderDetails_Products FOREIGN KEY (ProductId) 
        REFERENCES Products(ProductId)
);

-- Create Independent Table
CREATE TABLE LogEvents (
    LogId INT IDENTITY(1,1) PRIMARY KEY,
    EventType NVARCHAR(50) NOT NULL,
    EventMessage NVARCHAR(MAX),
    CreatedDate DATETIME DEFAULT GETDATE()
);

-- Create Indexes
CREATE NONCLUSTERED INDEX IX_Products_CategoryId ON Products(CategoryId);
CREATE NONCLUSTERED INDEX IX_OrderDetails_ProductId ON OrderDetails(ProductId);
CREATE NONCLUSTERED INDEX IX_LogEvents_EventType ON LogEvents(EventType);

-- Create Table Type for TVP
CREATE TYPE OrderDetailsType AS TABLE
(
    ProductId INT,
    Quantity INT
);
GO

-- Create Store Procedures
CREATE PROCEDURE sp_GetProductsByCategory
    @CategoryId INT
AS
BEGIN
    SELECT p.ProductId, p.ProductName, p.UnitPrice, c.CategoryName
    FROM Products p
    INNER JOIN Categories c ON p.CategoryId = c.CategoryId
    WHERE c.CategoryId = @CategoryId;
END;
GO

CREATE PROCEDURE sp_CreateOrder
    @OrderDetails OrderDetailsType READONLY
AS
BEGIN
    INSERT INTO OrderDetails (ProductId, Quantity)
    SELECT ProductId, Quantity FROM @OrderDetails;
END;
GO

CREATE PROCEDURE sp_GetOrderSummary
    @StartDate DATETIME,
    @EndDate DATETIME
AS
BEGIN
    SELECT 
        p.ProductName,
        c.CategoryName,
        SUM(od.Quantity) as TotalQuantity
    FROM OrderDetails od
    INNER JOIN Products p ON od.ProductId = p.ProductId
    INNER JOIN Categories c ON p.CategoryId = c.CategoryId
    WHERE od.OrderDate BETWEEN @StartDate AND @EndDate
    GROUP BY p.ProductName, c.CategoryName;
END;
GO

-- Create Table Function
CREATE FUNCTION fn_GetProductRevenue
(
    @StartDate DATETIME,
    @EndDate DATETIME
)
RETURNS TABLE
AS
RETURN
(
    SELECT 
        p.ProductId,
        p.ProductName,
        SUM(od.Quantity * p.UnitPrice) as Revenue
    FROM Products p
    INNER JOIN OrderDetails od ON p.ProductId = od.ProductId
    WHERE od.OrderDate BETWEEN @StartDate AND @EndDate
    GROUP BY p.ProductId, p.ProductName
);
GO

-- Grant Permissions
GRANT EXECUTE ON sp_GetProductsByCategory TO TestRole;
GRANT EXECUTE ON sp_CreateOrder TO TestRole;
GRANT EXECUTE ON sp_GetOrderSummary TO TestRole;
GRANT SELECT ON fn_GetProductRevenue TO TestRole;
GO 