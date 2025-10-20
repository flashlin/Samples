using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CoverBetAccountEntityConfiguration : IEntityTypeConfiguration<CoverBetAccountEntity>
    {
        public void Configure(EntityTypeBuilder<CoverBetAccountEntity> builder)
        {
            builder.ToTable("CoverBetAccount");

            builder.HasKey(x => x.CustID);

            builder.Property(x => x.CustID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.UserName)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Status)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("nvarchar(255)")
                .HasMaxLength(255)
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
            ;

        }
    }
}
