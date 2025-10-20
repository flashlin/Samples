using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MoneyTransferBankInfoEntityConfiguration : IEntityTypeConfiguration<MoneyTransferBankInfoEntity>
    {
        public void Configure(EntityTypeBuilder<MoneyTransferBankInfoEntity> builder)
        {
            builder.ToTable("MoneyTransferBankInfo");

            builder.HasKey(x => x.MTBID);

            builder.Property(x => x.MTBID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CustId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.ISOCUrrency)
                .HasColumnType("char(3)")
                .HasMaxLength(3)
            ;

            builder.Property(x => x.AccountId)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.AccountHolderName)
                .HasColumnType("nvarchar(100)")
                .HasMaxLength(100)
            ;

            builder.Property(x => x.AccountNumber)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.BankName)
                .HasColumnType("nvarchar(100)")
                .HasMaxLength(100)
            ;

            builder.Property(x => x.Branch)
                .HasColumnType("nvarchar(200)")
                .HasMaxLength(200)
            ;

            builder.Property(x => x.BankAddress)
                .HasColumnType("nvarchar(500)")
                .HasMaxLength(500)
            ;

            builder.Property(x => x.IBAN)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.SWIFT)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.SortCode)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.LastUsedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.BeneficiaryAddress)
                .HasColumnType("nvarchar(200)")
                .HasMaxLength(200)
            ;

        }
    }
}
