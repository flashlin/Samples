using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class CustomerRiskLimitEntityConfiguration : IEntityTypeConfiguration<CustomerRiskLimitEntity>
    {
        public void Configure(EntityTypeBuilder<CustomerRiskLimitEntity> builder)
        {
            builder.ToTable("CustomerRiskLimit");

            builder.HasKey(x => x.CustID);

            builder.Property(x => x.CustID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.ParentID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.RoleID)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.WinRisk)
                .HasColumnType("")
            ;

            builder.Property(x => x.LostRisk)
                .HasColumnType("")
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedDate)
                .HasColumnType("datetime")
            ;

        }
    }
}
