using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class APIAccountEntityConfiguration : IEntityTypeConfiguration<APIAccountEntity>
    {
        public void Configure(EntityTypeBuilder<APIAccountEntity> builder)
        {
            builder.ToTable("APIAccount");

            builder.HasKey(x => x.custid);

            builder.Property(x => x.custid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.username)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.company_name)
                .HasColumnType("nvarchar(100)")
                .HasMaxLength(100)
            ;

            builder.Property(x => x.ips)
                .HasColumnType("nvarchar(500)")
                .HasMaxLength(500)
            ;

            builder.Property(x => x.remark)
                .HasColumnType("nvarchar(100)")
                .HasMaxLength(100)
            ;

            builder.Property(x => x.status)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(1)
            ;

            builder.Property(x => x.modifiedby)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.modifieddate)
                .HasColumnType("datetime")
            ;

        }
    }
}
