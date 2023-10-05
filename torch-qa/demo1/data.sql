DROP TABLE IF EXISTS `Customers`;
CREATE TABLE `Customers`
(
    `Id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
    `LoginName` varchar(255) NOT NULL,
    `CreateOn` DATETIME NOT NULL,
    PRIMARY KEY (Id),
    UNIQUE KEY `UNX_Customers` (`LoginName`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `Customers` (LoginName, CreateOn)
    VALUES ('flash', NOW());


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
    `Question` text NOT NULL,
    `Answer` text NOT NULL,
    `CreateOn` DATETIME NOT NULL,
    PRIMARY KEY (Id),
    FOREIGN KEY (ConversationsId) REFERENCES Conversations(Id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;