DROP TABLE IF EXISTS `Customers`;
CREATE TABLE `Customers`
(
    `Id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
    `LoginName` varchar(255) NOT NULL,
    `Password` varchar(255) NOT NULL,
    `CreateOn` DATETIME NOT NULL,
    PRIMARY KEY (Id),
    UNIQUE KEY `UNX_Customers` (`LoginName`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `Customers` (LoginName, Password, CreateOn)
    VALUES ('flash', '$2b$12$77k0fAPlRF7tZ8zBu2a4D.ZCobB/iPVcIjxcOrYJ2i3aRDse64/p2', NOW());


CREATE TABLE `Conversations`
(
    `Id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
    `LoginName` varchar(255) NOT NULL,
    `CreateOn` DATETIME NOT NULL,
    PRIMARY KEY (Id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `ConversationsDetail`
(
    `Id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
    `ConversationsId` INT UNSIGNED NOT NULL,
    `RoleName` varchar(80) NOT NULL,
    `Message` text NOT NULL,
    `CreateOn` DATETIME NOT NULL,
    PRIMARY KEY (Id),
    FOREIGN KEY (ConversationsId) REFERENCES Conversations(Id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;