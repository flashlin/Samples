using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class WongLaiAccountEntityConfiguration : IEntityTypeConfiguration<WongLaiAccountEntity>
    {
        public void Configure(EntityTypeBuilder<WongLaiAccountEntity> builder)
        {
            builder.ToTable("WongLaiAccount");

            builder.HasKey(x => new { x.Currency, x.Type });

            builder.Property(x => x.Currency)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.Type)
                .HasColumnType("tinyint(3,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.CurrencyStr)
                .HasColumnType("varchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.TypeStr)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CustID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.AgentID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.MaID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.SmaID)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.UserName)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.status)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

            builder.Property(x => x.tstamp)
                .HasColumnType("timestamp")
            ;

        }
    }
}
