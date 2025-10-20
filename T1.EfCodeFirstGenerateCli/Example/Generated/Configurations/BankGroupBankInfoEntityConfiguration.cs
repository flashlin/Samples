using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class BankGroupBankInfoEntityConfiguration : IEntityTypeConfiguration<BankGroupBankInfoEntity>
    {
        public void Configure(EntityTypeBuilder<BankGroupBankInfoEntity> builder)
        {
            builder.ToTable("BankGroupBankInfo");


            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.AccountId)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CompanyName)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.BeneficiaryAddress)
                .HasColumnType("nvarchar(300)")
                .IsRequired()
                .HasMaxLength(300)
            ;

            builder.Property(x => x.BankName)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.BankCode)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.BankAddress)
                .HasColumnType("nvarchar(300)")
                .IsRequired()
                .HasMaxLength(300)
            ;

            builder.Property(x => x.BankSwiftCode)
                .HasColumnType("nvarchar(20)")
                .IsRequired()
                .HasMaxLength(20)
            ;

            builder.Property(x => x.Currency)
                .HasColumnType("nvarchar(5)")
                .IsRequired()
                .HasMaxLength(5)
            ;

            builder.Property(x => x.AccountNumber)
                .HasColumnType("nvarchar(30)")
                .IsRequired()
                .HasMaxLength(30)
            ;

            builder.Property(x => x.CorrespondentBank)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CorrespondentBankSwiftcode)
                .HasColumnType("nvarchar(20)")
                .IsRequired()
                .HasMaxLength(20)
            ;

            builder.Property(x => x.IBAN)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.SortCode)
                .HasColumnType("varchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.RoutingCode)
                .HasColumnType("varchar(30)")
                .HasMaxLength(30)
            ;

            builder.Property(x => x.Status)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.UpdatedBy)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.UpdatedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.Jurisdiction)
                .HasColumnType("varchar(20)")
                .HasMaxLength(20)
            ;

            builder.Property(x => x.BankAccountType)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.BankBranch)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.SlipDetails)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.DisplayName)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
