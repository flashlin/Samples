using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class BankAccountEntityConfiguration : IEntityTypeConfiguration<BankAccountEntity>
    {
        public void Configure(EntityTypeBuilder<BankAccountEntity> builder)
        {
            builder.ToTable("BankAccount");

            builder.HasKey(x => x.custid);

            builder.Property(x => x.custid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.username)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.type)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.description)
                .HasColumnType("nvarchar(150)")
                .HasMaxLength(150)
            ;

            builder.Property(x => x.currency)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.tstamp)
                .HasColumnType("timestamp")
            ;

        }
    }
}
